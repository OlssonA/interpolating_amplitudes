module     p0_gg_gh_d11h0l1_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d11h0l1_qp.f90
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
      use p0_gg_gh_abbrevd11h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc11(14)
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1k3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak1l4
      complex(ki) :: QspQ
      complex(ki) :: Qspk1
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      QspQ = dotproduct(Q,Q)
      Qspk1 = dotproduct(Q,k1)
      acc11(1)=abb11(8)
      acc11(2)=abb11(9)
      acc11(3)=abb11(10)
      acc11(4)=abb11(11)
      acc11(5)=abb11(12)
      acc11(6)=abb11(13)
      acc11(7)=abb11(14)
      acc11(8)=abb11(15)
      acc11(9)=Qspvak2k3*acc11(3)
      acc11(9)=acc11(9)+acc11(6)
      acc11(9)=Qspvak1k2*acc11(9)
      acc11(10)=acc11(8)*Qspvak1k3
      acc11(11)=acc11(7)*Qspval4k2
      acc11(12)=acc11(4)*Qspk2
      acc11(13)=acc11(1)*Qspvak1l4
      acc11(14)=QspQ+Qspk1
      acc11(14)=acc11(5)*acc11(14)
      acc11(9)=acc11(14)+acc11(13)+acc11(2)+acc11(12)+acc11(10)+acc11(11)+acc11&
      &(9)
      brack=Qspvak2k3*acc11(9)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d11h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd11h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d11
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d11 = 0.0_ki
      d11 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d11, ki), aimag(d11), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d11h0l1_qp
