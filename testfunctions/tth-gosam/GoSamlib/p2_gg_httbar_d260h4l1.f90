module     p2_gg_httbar_d260h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d260h4l1.f90
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
      use p2_gg_httbar_abbrevd260h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc260(107)
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspe2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval4e1
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspe2 = dotproduct(Q,e2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      QspQ = dotproduct(Q,Q)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      acc260(1)=abb260(8)
      acc260(2)=abb260(9)
      acc260(3)=abb260(10)
      acc260(4)=abb260(11)
      acc260(5)=abb260(12)
      acc260(6)=abb260(13)
      acc260(7)=abb260(14)
      acc260(8)=abb260(15)
      acc260(9)=abb260(16)
      acc260(10)=abb260(17)
      acc260(11)=abb260(18)
      acc260(12)=abb260(19)
      acc260(13)=abb260(20)
      acc260(14)=abb260(21)
      acc260(15)=abb260(22)
      acc260(16)=abb260(23)
      acc260(17)=abb260(24)
      acc260(18)=abb260(25)
      acc260(19)=abb260(26)
      acc260(20)=abb260(27)
      acc260(21)=abb260(28)
      acc260(22)=abb260(29)
      acc260(23)=abb260(30)
      acc260(24)=abb260(31)
      acc260(25)=abb260(32)
      acc260(26)=abb260(33)
      acc260(27)=abb260(34)
      acc260(28)=abb260(35)
      acc260(29)=abb260(36)
      acc260(30)=abb260(37)
      acc260(31)=abb260(38)
      acc260(32)=abb260(39)
      acc260(33)=abb260(40)
      acc260(34)=abb260(41)
      acc260(35)=abb260(42)
      acc260(36)=abb260(43)
      acc260(37)=abb260(44)
      acc260(38)=abb260(45)
      acc260(39)=abb260(46)
      acc260(40)=abb260(47)
      acc260(41)=abb260(48)
      acc260(42)=abb260(49)
      acc260(43)=abb260(50)
      acc260(44)=abb260(51)
      acc260(45)=abb260(52)
      acc260(46)=abb260(53)
      acc260(47)=abb260(54)
      acc260(48)=abb260(55)
      acc260(49)=abb260(56)
      acc260(50)=abb260(57)
      acc260(51)=abb260(58)
      acc260(52)=abb260(59)
      acc260(53)=abb260(60)
      acc260(54)=abb260(61)
      acc260(55)=abb260(62)
      acc260(56)=abb260(63)
      acc260(57)=abb260(64)
      acc260(58)=abb260(65)
      acc260(59)=abb260(66)
      acc260(60)=abb260(67)
      acc260(61)=abb260(68)
      acc260(62)=abb260(69)
      acc260(63)=abb260(70)
      acc260(64)=abb260(71)
      acc260(65)=abb260(72)
      acc260(66)=abb260(73)
      acc260(67)=abb260(75)
      acc260(68)=abb260(77)
      acc260(69)=abb260(78)
      acc260(70)=abb260(79)
      acc260(71)=abb260(81)
      acc260(72)=abb260(86)
      acc260(73)=abb260(94)
      acc260(74)=abb260(96)
      acc260(75)=abb260(99)
      acc260(76)=abb260(109)
      acc260(77)=abb260(111)
      acc260(78)=abb260(112)
      acc260(79)=abb260(118)
      acc260(80)=abb260(122)
      acc260(81)=abb260(123)
      acc260(82)=abb260(129)
      acc260(83)=abb260(134)
      acc260(84)=abb260(135)
      acc260(85)=abb260(136)
      acc260(86)=abb260(141)
      acc260(87)=abb260(142)
      acc260(88)=Qspvae1l4*acc260(44)
      acc260(89)=Qspval5e1*acc260(74)
      acc260(90)=Qspvae1k2*acc260(38)
      acc260(91)=Qspvak2e1*acc260(27)
      acc260(88)=acc260(91)+acc260(90)+acc260(89)+acc260(45)+acc260(88)
      acc260(88)=Qspe2*acc260(88)
      acc260(89)=Qspvak1l4*acc260(65)
      acc260(90)=Qspvak2k1*acc260(59)
      acc260(91)=Qspvae1l5*acc260(85)
      acc260(92)=Qspvae1l4*acc260(56)
      acc260(93)=Qspval5e1*acc260(54)
      acc260(94)=Qspvae1k2*acc260(19)
      acc260(95)=Qspvae1l3*acc260(73)
      acc260(96)=Qspval3e1*acc260(51)
      acc260(97)=Qspval5e1*acc260(64)
      acc260(97)=acc260(66)+acc260(97)
      acc260(97)=Qspvae1e2*acc260(97)
      acc260(98)=Qspvae1k2*acc260(72)
      acc260(98)=acc260(12)+acc260(98)
      acc260(98)=Qspvae2e1*acc260(98)
      acc260(99)=Qspvak2e1*acc260(17)
      acc260(100)=QspQ*acc260(63)
      acc260(88)=acc260(100)+acc260(88)+acc260(99)+acc260(98)+acc260(97)+acc260&
      &(96)+acc260(95)+acc260(94)+acc260(93)+acc260(92)+acc260(91)+acc260(90)+a&
      &cc260(35)+acc260(89)
      acc260(88)=QspQ*acc260(88)
      acc260(89)=Qspvae1k1*acc260(15)
      acc260(90)=Qspval5k2*acc260(55)
      acc260(91)=Qspval5l3*acc260(40)
      acc260(92)=Qspvak2k1*acc260(83)
      acc260(93)=Qspvae1l5*acc260(86)
      acc260(94)=Qspvae1l4*acc260(75)
      acc260(95)=Qspk2*acc260(49)
      acc260(95)=acc260(9)+acc260(95)
      acc260(95)=Qspvae1k2*acc260(95)
      acc260(96)=-Qspval3k2*acc260(72)
      acc260(96)=acc260(80)+acc260(96)
      acc260(96)=Qspvae1l3*acc260(96)
      acc260(89)=acc260(96)+acc260(95)+acc260(94)+acc260(93)+acc260(92)+acc260(&
      &91)+acc260(90)+acc260(26)+acc260(89)
      acc260(89)=Qspvae2e1*acc260(89)
      acc260(90)=Qspvak1e1*acc260(28)
      acc260(91)=Qspval4e1*acc260(78)
      acc260(92)=Qspk2*acc260(22)
      acc260(93)=-Qspvak1l4*acc260(47)
      acc260(94)=Qspval3k2*acc260(81)
      acc260(95)=Qspval5e1*acc260(6)
      acc260(96)=-Qspval5l3*acc260(64)
      acc260(96)=acc260(1)+acc260(96)
      acc260(96)=Qspval3e1*acc260(96)
      acc260(90)=acc260(96)+acc260(95)+acc260(94)+acc260(93)+acc260(92)+acc260(&
      &91)+acc260(5)+acc260(90)
      acc260(90)=Qspvae1e2*acc260(90)
      acc260(91)=Qspval5k2*acc260(23)
      acc260(92)=Qspvak2k1*acc260(50)
      acc260(93)=Qspvae1l5*acc260(48)
      acc260(94)=Qspvae1l4*acc260(53)
      acc260(95)=Qspvae1k2*acc260(11)
      acc260(96)=Qspvae1l3*acc260(16)
      acc260(97)=Qspval5k2*acc260(20)
      acc260(97)=acc260(32)+acc260(97)
      acc260(97)=Qspvae1e2*acc260(97)
      acc260(91)=acc260(97)+acc260(96)+acc260(95)+acc260(94)+acc260(93)+acc260(&
      &92)+acc260(7)+acc260(91)
      acc260(91)=Qspvak2e1*acc260(91)
      acc260(92)=Qspval5l3*acc260(41)
      acc260(93)=Qspvak2k1*acc260(60)
      acc260(94)=Qspvae1l5*acc260(68)
      acc260(95)=Qspvae1l4*acc260(42)
      acc260(96)=Qspvae1k2*acc260(61)
      acc260(97)=Qspvae1l3*acc260(69)
      acc260(92)=acc260(97)+acc260(96)+acc260(95)+acc260(94)+acc260(93)+acc260(&
      &31)+acc260(92)
      acc260(92)=Qspval3e1*acc260(92)
      acc260(93)=Qspvae1l4*acc260(87)
      acc260(94)=Qspvae1k2*acc260(70)
      acc260(93)=acc260(94)+acc260(43)+acc260(93)
      acc260(93)=Qspval3e1*acc260(93)
      acc260(94)=Qspvae1k2*acc260(3)
      acc260(95)=Qspvae1l3*acc260(25)
      acc260(94)=acc260(95)+acc260(13)+acc260(94)
      acc260(94)=Qspvak2e1*acc260(94)
      acc260(95)=Qspvae1l4*acc260(39)
      acc260(96)=Qspvae1l4*acc260(77)
      acc260(96)=acc260(52)+acc260(96)
      acc260(96)=Qspval5e1*acc260(96)
      acc260(97)=Qspvae1k2*acc260(18)
      acc260(98)=Qspval5e1*acc260(84)
      acc260(98)=acc260(67)+acc260(98)
      acc260(98)=Qspvae1l3*acc260(98)
      acc260(93)=acc260(94)+acc260(93)+acc260(98)+acc260(97)+acc260(96)+acc260(&
      &2)+acc260(95)
      acc260(93)=Qspe2*acc260(93)
      acc260(94)=Qspvak1l4*acc260(24)
      acc260(95)=Qspval3k2*acc260(34)
      acc260(96)=Qspval5e1*acc260(82)
      acc260(94)=acc260(96)+acc260(95)+acc260(21)+acc260(94)
      acc260(94)=Qspvae1l3*acc260(94)
      acc260(95)=Qspvak1e1*acc260(29)
      acc260(96)=-Qspvae1k1*acc260(58)
      acc260(97)=Qspval4e1*acc260(36)
      acc260(98)=Qspk2*acc260(79)
      acc260(99)=Qspvak1l4*acc260(62)
      acc260(100)=Qspval3k2*acc260(33)
      acc260(101)=Qspval5k2*acc260(46)
      acc260(102)=Qspval5l3*acc260(37)
      acc260(103)=Qspvak2k1*acc260(57)
      acc260(104)=Qspvae1l5*acc260(71)
      acc260(105)=Qspvae1l4*acc260(4)
      acc260(106)=-Qspvae1l4*acc260(76)
      acc260(106)=acc260(10)+acc260(106)
      acc260(106)=Qspval5e1*acc260(106)
      acc260(107)=-Qspk2*acc260(30)
      acc260(107)=acc260(14)+acc260(107)
      acc260(107)=Qspvae1k2*acc260(107)
      brack=acc260(8)+acc260(88)+acc260(89)+acc260(90)+acc260(91)+acc260(92)+ac&
      &c260(93)+acc260(94)+acc260(95)+acc260(96)+acc260(97)+acc260(98)+acc260(9&
      &9)+acc260(100)+acc260(101)+acc260(102)+acc260(103)+acc260(104)+acc260(10&
      &5)+acc260(106)+acc260(107)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d260h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd260h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d260
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d260 = 0.0_ki
      d260 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d260, ki), aimag(d260), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d260h4l1
