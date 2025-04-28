module     p0_gg_gh_d7h4l1_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d7h4l1_qp.f90
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
      use p0_gg_gh_abbrevd7h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc7(19)
      complex(ki) :: Qspk2
      complex(ki) :: Qspk3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1k3
      complex(ki) :: QspQ
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak3k2
      complex(ki) :: Qspvak2k1
      Qspk2 = dotproduct(Q,k2)
      Qspk3 = dotproduct(Q,k3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      QspQ = dotproduct(Q,Q)
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      acc7(1)=abb7(5)
      acc7(2)=abb7(6)
      acc7(3)=abb7(7)
      acc7(4)=abb7(8)
      acc7(5)=abb7(9)
      acc7(6)=abb7(10)
      acc7(7)=abb7(11)
      acc7(8)=abb7(12)
      acc7(9)=abb7(13)
      acc7(10)=abb7(14)
      acc7(11)=abb7(15)
      acc7(12)=abb7(16)
      acc7(13)=abb7(17)
      acc7(14)=Qspk2+Qspk3
      acc7(15)=-acc7(13)*acc7(14)
      acc7(16)=Qspvak1k2*acc7(4)
      acc7(17)=Qspvak1k3*acc7(1)
      acc7(15)=acc7(17)+acc7(16)+acc7(12)+acc7(15)
      acc7(15)=Qspk2*acc7(15)
      acc7(14)=-QspQ+acc7(14)
      acc7(14)=acc7(3)*acc7(14)
      acc7(16)=Qspvak1k2*acc7(6)*Qspvak2k3
      acc7(17)=Qspvak1k3*acc7(7)
      acc7(14)=acc7(17)+acc7(2)+acc7(16)+acc7(14)
      acc7(14)=Qspvak3k2*acc7(14)
      acc7(16)=QspQ*acc7(10)
      acc7(17)=Qspk3*acc7(11)
      acc7(18)=Qspvak1k2*acc7(8)*Qspvak2k1
      acc7(19)=Qspvak1k3*acc7(9)
      brack=acc7(5)+acc7(14)+acc7(15)+acc7(16)+acc7(17)+acc7(18)+acc7(19)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d7h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd7h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d7
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d7 = 0.0_ki
      d7 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d7, ki), aimag(d7), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d7h4l1_qp
