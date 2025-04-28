module     p2_gg_httbar_d254h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d254h8l1.f90
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
      use p2_gg_httbar_abbrevd254h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc254(95)
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k1
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      QspQ = dotproduct(Q,Q)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      acc254(1)=abb254(7)
      acc254(2)=abb254(8)
      acc254(3)=abb254(9)
      acc254(4)=abb254(10)
      acc254(5)=abb254(11)
      acc254(6)=abb254(12)
      acc254(7)=abb254(13)
      acc254(8)=abb254(14)
      acc254(9)=abb254(15)
      acc254(10)=abb254(16)
      acc254(11)=abb254(17)
      acc254(12)=abb254(18)
      acc254(13)=abb254(19)
      acc254(14)=abb254(20)
      acc254(15)=abb254(21)
      acc254(16)=abb254(22)
      acc254(17)=abb254(23)
      acc254(18)=abb254(24)
      acc254(19)=abb254(25)
      acc254(20)=abb254(26)
      acc254(21)=abb254(27)
      acc254(22)=abb254(28)
      acc254(23)=abb254(29)
      acc254(24)=abb254(30)
      acc254(25)=abb254(31)
      acc254(26)=abb254(32)
      acc254(27)=abb254(33)
      acc254(28)=abb254(34)
      acc254(29)=abb254(35)
      acc254(30)=abb254(36)
      acc254(31)=abb254(37)
      acc254(32)=abb254(38)
      acc254(33)=abb254(39)
      acc254(34)=abb254(40)
      acc254(35)=abb254(41)
      acc254(36)=abb254(42)
      acc254(37)=abb254(43)
      acc254(38)=abb254(44)
      acc254(39)=abb254(45)
      acc254(40)=abb254(46)
      acc254(41)=abb254(47)
      acc254(42)=abb254(48)
      acc254(43)=abb254(49)
      acc254(44)=abb254(50)
      acc254(45)=abb254(51)
      acc254(46)=abb254(52)
      acc254(47)=abb254(53)
      acc254(48)=abb254(54)
      acc254(49)=abb254(55)
      acc254(50)=abb254(56)
      acc254(51)=abb254(58)
      acc254(52)=abb254(59)
      acc254(53)=abb254(61)
      acc254(54)=abb254(62)
      acc254(55)=abb254(63)
      acc254(56)=abb254(64)
      acc254(57)=abb254(65)
      acc254(58)=abb254(66)
      acc254(59)=abb254(68)
      acc254(60)=abb254(69)
      acc254(61)=abb254(70)
      acc254(62)=abb254(71)
      acc254(63)=abb254(72)
      acc254(64)=abb254(73)
      acc254(65)=abb254(74)
      acc254(66)=abb254(75)
      acc254(67)=abb254(76)
      acc254(68)=abb254(77)
      acc254(69)=abb254(78)
      acc254(70)=-acc254(31)*Qspval4k1
      acc254(71)=Qspvak2k1*acc254(1)
      acc254(70)=acc254(71)+acc254(60)+acc254(70)
      acc254(70)=Qspvak1e1*acc254(70)
      acc254(71)=acc254(31)*Qspval4l3
      acc254(72)=-acc254(1)*Qspvak2l3
      acc254(71)=acc254(72)+acc254(53)+acc254(71)
      acc254(71)=Qspval3e1*acc254(71)
      acc254(72)=Qspval4k2*acc254(65)
      acc254(73)=Qspk2*acc254(5)
      acc254(72)=acc254(73)+acc254(23)+acc254(72)
      acc254(72)=Qspvak2e1*acc254(72)
      acc254(73)=acc254(31)*Qspval4e1
      acc254(74)=-Qspvak2e1*acc254(1)
      acc254(73)=acc254(74)+acc254(54)+acc254(73)
      acc254(73)=QspQ*acc254(73)
      acc254(74)=Qspvak1k2*acc254(26)
      acc254(75)=Qspvak2l3*acc254(34)
      acc254(76)=Qspval4k2*acc254(48)
      acc254(77)=Qspval4l3*acc254(63)
      acc254(78)=Qspval4e1*acc254(56)
      acc254(79)=Qspvak1l5*acc254(18)
      acc254(80)=Qspk2*acc254(52)
      acc254(81)=Qspvak2e1*acc254(3)
      acc254(81)=acc254(47)+acc254(81)
      acc254(81)=Qspvae2k2*acc254(81)
      acc254(70)=acc254(73)+acc254(81)+acc254(72)+acc254(71)+acc254(70)+acc254(&
      &80)+acc254(79)+acc254(78)+acc254(77)+acc254(76)+acc254(75)+acc254(39)+ac&
      &c254(74)
      acc254(70)=Qspvae1e2*acc254(70)
      acc254(71)=-acc254(25)*Qspvak1k2
      acc254(72)=Qspvak1l5*acc254(32)
      acc254(71)=acc254(72)+acc254(10)+acc254(71)
      acc254(71)=Qspvae1k1*acc254(71)
      acc254(72)=Qspvak2l5*acc254(8)
      acc254(73)=Qspk2*acc254(30)
      acc254(72)=acc254(73)+acc254(51)+acc254(72)
      acc254(72)=Qspvae1k2*acc254(72)
      acc254(73)=-acc254(32)*Qspval3l5
      acc254(74)=acc254(25)*Qspval3k2
      acc254(73)=acc254(74)+acc254(57)+acc254(73)
      acc254(73)=Qspvae1l3*acc254(73)
      acc254(74)=-Qspvae1l5*acc254(32)
      acc254(75)=Qspvae1k2*acc254(25)
      acc254(74)=acc254(75)+acc254(45)+acc254(74)
      acc254(74)=QspQ*acc254(74)
      acc254(75)=Qspvak2l5*acc254(46)
      acc254(76)=Qspval3k2*acc254(49)
      acc254(77)=Qspval3l5*acc254(67)
      acc254(78)=-Qspvak2k1*acc254(38)
      acc254(79)=Qspvae1l5*acc254(11)
      acc254(80)=Qspk2*acc254(44)
      acc254(81)=Qspvae1l5*acc254(15)
      acc254(81)=acc254(42)+acc254(81)
      acc254(81)=Qspval4e2*acc254(81)
      acc254(71)=acc254(74)+acc254(81)+acc254(73)+acc254(72)+acc254(71)+acc254(&
      &80)+acc254(79)+acc254(78)+acc254(77)+acc254(76)+acc254(22)+acc254(75)
      acc254(71)=Qspvae2e1*acc254(71)
      acc254(72)=Qspvak1l5*acc254(16)
      acc254(73)=Qspvak1e1*acc254(61)
      acc254(74)=Qspval3e1*acc254(58)
      acc254(75)=Qspvak2e1*acc254(40)
      acc254(72)=acc254(75)+acc254(74)+acc254(73)+acc254(2)+acc254(72)
      acc254(72)=Qspval4e2*acc254(72)
      acc254(73)=Qspvak2k1*acc254(37)
      acc254(74)=Qspvae1k1*acc254(55)
      acc254(75)=-Qspvae1l3*acc254(59)
      acc254(73)=acc254(75)+acc254(74)+acc254(21)+acc254(73)
      acc254(73)=Qspvae2k2*acc254(73)
      acc254(74)=Qspvae2k2*acc254(41)
      acc254(75)=Qspval4e2*acc254(17)
      acc254(74)=acc254(75)+acc254(28)+acc254(74)
      acc254(74)=QspQ*acc254(74)
      acc254(75)=acc254(50)*Qspvak1e2
      acc254(76)=acc254(24)*Qspvae2k1
      acc254(77)=Qspval4k1*acc254(35)
      acc254(78)=Qspvak1k2*acc254(13)
      acc254(79)=Qspvak2l3*acc254(69)
      acc254(80)=Qspvak2l5*acc254(12)
      acc254(81)=Qspval3k2*acc254(68)
      acc254(82)=Qspval3l5*acc254(66)
      acc254(83)=Qspval4k2*acc254(64)
      acc254(84)=Qspval4l3*acc254(62)
      acc254(85)=Qspval4e1*acc254(33)
      acc254(86)=Qspvak1l5*acc254(14)
      acc254(87)=Qspvak2k1*acc254(6)
      acc254(88)=Qspvae1l5*acc254(43)
      acc254(89)=Qspk2*acc254(29)
      acc254(90)=Qspvak1e1*acc254(4)
      acc254(91)=Qspvae1k1*acc254(20)
      acc254(92)=Qspvae1k2*acc254(36)
      acc254(93)=Qspval3e1*acc254(7)
      acc254(94)=Qspvae1l3*acc254(27)
      acc254(95)=Qspvak2e1*acc254(19)
      brack=acc254(9)+acc254(70)+acc254(71)+acc254(72)+acc254(73)+acc254(74)+ac&
      &c254(75)+acc254(76)+acc254(77)+acc254(78)+acc254(79)+acc254(80)+acc254(8&
      &1)+acc254(82)+acc254(83)+acc254(84)+acc254(85)+acc254(86)+acc254(87)+acc&
      &254(88)+acc254(89)+acc254(90)+acc254(91)+acc254(92)+acc254(93)+acc254(94&
      &)+acc254(95)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d254h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd254h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d254
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d254 = 0.0_ki
      d254 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d254, ki), aimag(d254), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d254h8l1
