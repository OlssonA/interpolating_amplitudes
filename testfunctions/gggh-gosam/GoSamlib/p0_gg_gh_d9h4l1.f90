module     p0_gg_gh_d9h4l1
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d9h4l1.f90
   ! generator: buildfortran.py
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_gg_gh_model
      use p0_gg_gh_kinematics
      use p0_gg_gh_color
      use p0_gg_gh_abbrevd9h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc9(21)
      complex(ki) :: Qspvak1k3
      complex(ki) :: Qspk3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4k2
      complex(ki) :: QspQ
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak3k2
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      Qspk3 = dotproduct(Q,k3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      QspQ = dotproduct(Q,Q)
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      acc9(1)=abb9(5)
      acc9(2)=abb9(6)
      acc9(3)=abb9(7)
      acc9(4)=abb9(8)
      acc9(5)=abb9(9)
      acc9(6)=abb9(10)
      acc9(7)=abb9(11)
      acc9(8)=abb9(12)
      acc9(9)=abb9(13)
      acc9(10)=abb9(14)
      acc9(11)=abb9(15)
      acc9(12)=abb9(16)
      acc9(13)=abb9(18)
      acc9(14)=abb9(19)
      acc9(15)=abb9(21)
      acc9(16)=acc9(2)*Qspvak1k3
      acc9(17)=acc9(11)*Qspk3
      acc9(18)=acc9(13)*Qspvak1k2
      acc9(16)=acc9(18)+acc9(17)+acc9(10)+acc9(16)
      acc9(16)=Qspk2*acc9(16)
      acc9(17)=acc9(6)*Qspvak1k3
      acc9(18)=acc9(15)*Qspk3
      acc9(17)=acc9(18)+acc9(14)+acc9(17)
      acc9(17)=Qspval4k2*acc9(17)
      acc9(18)=acc9(1)*QspQ
      acc9(19)=Qspvak2k3*acc9(8)*Qspvak1k2
      acc9(18)=acc9(19)+acc9(12)+acc9(18)
      acc9(18)=Qspvak3k2*acc9(18)
      acc9(19)=acc9(5)*QspQ
      acc9(19)=acc9(19)+acc9(3)
      acc9(19)=Qspvak1k2*acc9(19)
      acc9(20)=acc9(4)*Qspvak1k3
      acc9(21)=acc9(7)*Qspk3
      brack=acc9(9)+acc9(16)+acc9(17)+acc9(18)+acc9(19)+acc9(20)+acc9(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d9h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd9h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d9
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d9 = 0.0_ki
      d9 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d9, ki), aimag(d9), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_gg_gh_d9h4l1
