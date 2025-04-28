module     p0_gg_gh_d1h1l1_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity1d1h1l1_qp.f90
   ! generator: buildfortran.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd1h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc1(8)
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l4
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      acc1(1)=abb1(6)
      acc1(2)=abb1(7)
      acc1(3)=abb1(8)
      acc1(4)=abb1(9)
      acc1(5)=abb1(12)
      acc1(6)=acc1(1)*Qspvak2k3
      acc1(6)=acc1(6)+acc1(2)
      acc1(6)=Qspvak2k1*acc1(6)
      acc1(7)=acc1(3)*Qspvak2k3
      acc1(8)=Qspvak2l4*acc1(4)
      brack=acc1(5)+acc1(6)+acc1(7)+acc1(8)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d1h1l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd1h1_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d1
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d1 = 0.0_ki
      d1 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d1, ki), aimag(d1), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d1h1l1_qp
