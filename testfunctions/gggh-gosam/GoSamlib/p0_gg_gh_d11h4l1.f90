module     p0_gg_gh_d11h4l1
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity4d11h4l1.f90
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
      use p0_gg_gh_abbrevd11h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc11(27)
      complex(ki) :: Qspk3
      complex(ki) :: QspQ
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k3
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak3k2
      complex(ki) :: Qspvak2k3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspk1
      Qspk3 = dotproduct(Q,k3)
      QspQ = dotproduct(Q,Q)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k3 = dotproduct(Q,spvak1k3)
      Qspk2 = dotproduct(Q,k2)
      Qspvak3k2 = dotproduct(Q,spvak3k2)
      Qspvak2k3 = dotproduct(Q,spvak2k3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspk1 = dotproduct(Q,k1)
      acc11(1)=abb11(7)
      acc11(2)=abb11(8)
      acc11(3)=abb11(9)
      acc11(4)=abb11(10)
      acc11(5)=abb11(11)
      acc11(6)=abb11(12)
      acc11(7)=abb11(13)
      acc11(8)=abb11(14)
      acc11(9)=abb11(15)
      acc11(10)=abb11(16)
      acc11(11)=abb11(17)
      acc11(12)=abb11(19)
      acc11(13)=abb11(20)
      acc11(14)=abb11(21)
      acc11(15)=abb11(23)
      acc11(16)=abb11(24)
      acc11(17)=abb11(25)
      acc11(18)=abb11(26)
      acc11(19)=abb11(27)
      acc11(20)=abb11(29)
      acc11(21)=acc11(12)*Qspk3
      acc11(22)=acc11(20)*QspQ
      acc11(23)=Qspval4k2*acc11(5)
      acc11(24)=Qspvak1k3*acc11(7)
      acc11(25)=Qspk2*acc11(6)
      acc11(26)=Qspvak3k2*acc11(1)*Qspvak2k3
      acc11(27)=Qspvak1k2*acc11(9)
      acc11(21)=acc11(27)+acc11(26)+acc11(25)+acc11(24)+acc11(23)+acc11(22)+acc&
      &11(21)+acc11(2)
      acc11(21)=Qspvak1k2*acc11(21)
      acc11(22)=acc11(15)*Qspvak1l4
      acc11(23)=acc11(3)*Qspk1
      acc11(24)=-Qspvak1k3*acc11(20)
      acc11(25)=Qspk2*acc11(13)
      acc11(22)=acc11(25)+acc11(24)+acc11(23)+acc11(22)+acc11(8)
      acc11(22)=Qspvak3k2*acc11(22)
      acc11(23)=Qspval4k2*acc11(19)
      acc11(24)=Qspvak1k3*acc11(18)
      acc11(25)=Qspk2*acc11(14)
      acc11(23)=acc11(25)+acc11(24)+acc11(10)+acc11(23)
      acc11(23)=Qspk2*acc11(23)
      acc11(24)=QspQ*acc11(16)
      acc11(25)=Qspval4k2*acc11(17)
      acc11(26)=Qspvak1k3*acc11(11)
      brack=acc11(4)+acc11(21)+acc11(22)+acc11(23)+acc11(24)+acc11(25)+acc11(26)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_gg_gh_d11h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_gg_gh_globalsl1, only: epspow
      use p0_gg_gh_kinematics
      use p0_gg_gh_abbrevd11h4
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
end module p0_gg_gh_d11h4l1
