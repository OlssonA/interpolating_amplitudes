module     p2_gg_httbar_d46h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d46h0l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd46h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc46(36)
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak2l3
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      QspQ = dotproduct(Q,Q)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      acc46(1)=abb46(9)
      acc46(2)=abb46(10)
      acc46(3)=abb46(11)
      acc46(4)=abb46(12)
      acc46(5)=abb46(13)
      acc46(6)=abb46(14)
      acc46(7)=abb46(15)
      acc46(8)=abb46(16)
      acc46(9)=abb46(17)
      acc46(10)=abb46(18)
      acc46(11)=abb46(19)
      acc46(12)=abb46(21)
      acc46(13)=abb46(23)
      acc46(14)=abb46(24)
      acc46(15)=abb46(25)
      acc46(16)=abb46(26)
      acc46(17)=abb46(27)
      acc46(18)=abb46(28)
      acc46(19)=abb46(32)
      acc46(20)=abb46(34)
      acc46(21)=abb46(35)
      acc46(22)=abb46(36)
      acc46(23)=abb46(37)
      acc46(24)=abb46(38)
      acc46(25)=abb46(43)
      acc46(26)=acc46(17)*Qspval4l3
      acc46(27)=acc46(12)*Qspval5l3
      acc46(26)=acc46(26)+acc46(27)
      acc46(27)=Qspval3k2*acc46(11)
      acc46(28)=Qspval4k2*acc46(2)
      acc46(29)=Qspval5k2*acc46(14)
      acc46(27)=acc46(29)+acc46(28)+acc46(27)+acc46(5)+acc46(26)
      acc46(27)=Qspk1*acc46(27)
      acc46(28)=Qspval3k2*acc46(10)
      acc46(29)=Qspval4k2*acc46(13)
      acc46(30)=Qspval5k2*acc46(16)
      acc46(26)=acc46(30)+acc46(29)+acc46(28)+acc46(6)-acc46(26)
      acc46(26)=Qspk2*acc46(26)
      acc46(28)=acc46(8)*Qspval3k1
      acc46(29)=Qspval4k1*acc46(7)
      acc46(30)=Qspval5k1*acc46(4)
      acc46(28)=acc46(30)+acc46(29)+acc46(28)+acc46(1)
      acc46(28)=Qspvak1k2*acc46(28)
      acc46(29)=acc46(9)*QspQ
      acc46(30)=Qspvak1l3*acc46(3)
      acc46(31)=Qspvak2l3*acc46(15)
      acc46(32)=Qspval3k2*acc46(20)
      acc46(33)=-Qspvak1l3*acc46(24)
      acc46(33)=acc46(19)+acc46(33)
      acc46(33)=Qspval4k1*acc46(33)
      acc46(34)=Qspvak1l3*acc46(23)
      acc46(34)=acc46(22)+acc46(34)
      acc46(34)=Qspval5k1*acc46(34)
      acc46(35)=Qspvak2l3*acc46(24)
      acc46(35)=acc46(18)+acc46(35)
      acc46(35)=Qspval4k2*acc46(35)
      acc46(36)=-Qspvak2l3*acc46(23)
      acc46(36)=acc46(21)+acc46(36)
      acc46(36)=Qspval5k2*acc46(36)
      brack=acc46(25)+acc46(26)+acc46(27)+acc46(28)+acc46(29)+acc46(30)+acc46(3&
      &1)+acc46(32)+acc46(33)+acc46(34)+acc46(35)+acc46(36)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d46h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd46h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d46
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k4-k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d46 = 0.0_ki
      d46 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d46, ki), aimag(d46), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d46h0l1
