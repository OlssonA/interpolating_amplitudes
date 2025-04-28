module     p2_gg_httbar_d82h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d82h8l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd82h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc82(116)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspk2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: QspQ
      complex(ki) :: Qspe2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspk1
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspk2 = dotproduct(Q,k2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspl5 = dotproduct(Q,l5)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      QspQ = dotproduct(Q,Q)
      Qspe2 = dotproduct(Q,e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspk1 = dotproduct(Q,k1)
      acc82(1)=abb82(8)
      acc82(2)=abb82(9)
      acc82(3)=abb82(10)
      acc82(4)=abb82(11)
      acc82(5)=abb82(12)
      acc82(6)=abb82(13)
      acc82(7)=abb82(14)
      acc82(8)=abb82(15)
      acc82(9)=abb82(16)
      acc82(10)=abb82(17)
      acc82(11)=abb82(18)
      acc82(12)=abb82(19)
      acc82(13)=abb82(20)
      acc82(14)=abb82(21)
      acc82(15)=abb82(22)
      acc82(16)=abb82(23)
      acc82(17)=abb82(25)
      acc82(18)=abb82(27)
      acc82(19)=abb82(28)
      acc82(20)=abb82(29)
      acc82(21)=abb82(30)
      acc82(22)=abb82(31)
      acc82(23)=abb82(32)
      acc82(24)=abb82(33)
      acc82(25)=abb82(34)
      acc82(26)=abb82(35)
      acc82(27)=abb82(36)
      acc82(28)=abb82(37)
      acc82(29)=abb82(38)
      acc82(30)=abb82(39)
      acc82(31)=abb82(40)
      acc82(32)=abb82(41)
      acc82(33)=abb82(42)
      acc82(34)=abb82(43)
      acc82(35)=abb82(44)
      acc82(36)=abb82(46)
      acc82(37)=abb82(47)
      acc82(38)=abb82(48)
      acc82(39)=abb82(49)
      acc82(40)=abb82(50)
      acc82(41)=abb82(51)
      acc82(42)=abb82(52)
      acc82(43)=abb82(53)
      acc82(44)=abb82(54)
      acc82(45)=abb82(55)
      acc82(46)=abb82(56)
      acc82(47)=abb82(57)
      acc82(48)=abb82(58)
      acc82(49)=abb82(61)
      acc82(50)=abb82(62)
      acc82(51)=abb82(63)
      acc82(52)=abb82(65)
      acc82(53)=abb82(66)
      acc82(54)=abb82(67)
      acc82(55)=abb82(68)
      acc82(56)=abb82(69)
      acc82(57)=abb82(70)
      acc82(58)=abb82(71)
      acc82(59)=abb82(72)
      acc82(60)=abb82(73)
      acc82(61)=abb82(74)
      acc82(62)=abb82(75)
      acc82(63)=abb82(76)
      acc82(64)=abb82(77)
      acc82(65)=abb82(78)
      acc82(66)=abb82(79)
      acc82(67)=abb82(81)
      acc82(68)=abb82(82)
      acc82(69)=abb82(84)
      acc82(70)=abb82(85)
      acc82(71)=abb82(86)
      acc82(72)=abb82(87)
      acc82(73)=abb82(88)
      acc82(74)=abb82(89)
      acc82(75)=abb82(90)
      acc82(76)=abb82(92)
      acc82(77)=abb82(93)
      acc82(78)=abb82(94)
      acc82(79)=abb82(95)
      acc82(80)=abb82(96)
      acc82(81)=abb82(97)
      acc82(82)=abb82(98)
      acc82(83)=abb82(99)
      acc82(84)=abb82(100)
      acc82(85)=abb82(101)
      acc82(86)=abb82(102)
      acc82(87)=abb82(103)
      acc82(88)=abb82(104)
      acc82(89)=abb82(105)
      acc82(90)=abb82(106)
      acc82(91)=abb82(108)
      acc82(92)=Qspvak2l3*acc82(91)
      acc82(93)=Qspval3l5*acc82(88)
      acc82(94)=Qspval4l5*acc82(80)
      acc82(95)=Qspk2*acc82(21)
      acc82(92)=acc82(95)+acc82(94)+acc82(93)+acc82(1)+acc82(92)
      acc82(92)=Qspe1*acc82(92)
      acc82(93)=acc82(75)*Qspvae1l3
      acc82(94)=acc82(68)*Qspval3e1
      acc82(95)=acc82(62)*Qspvae1l5
      acc82(96)=acc82(41)*Qspval4e1
      acc82(97)=acc82(33)*Qspvae1k1
      acc82(98)=acc82(24)*Qspvak2e1
      acc82(99)=acc82(20)*Qspvak1e1
      acc82(100)=acc82(18)*Qspvae1k2
      acc82(101)=Qspvak1k2*acc82(5)
      acc82(102)=Qspvak1l3*acc82(46)
      acc82(103)=Qspval3k1*acc82(73)
      acc82(104)=Qspval3k2*acc82(23)
      acc82(105)=Qspval4k1*acc82(30)
      acc82(106)=Qspval4k2*acc82(49)
      acc82(107)=Qspval5k2*acc82(58)
      acc82(108)=Qspval5l3*acc82(25)
      acc82(109)=Qspl5*acc82(7)
      acc82(110)=Qspvak1l5*acc82(19)
      acc82(111)=Qspvak2k1*acc82(66)
      acc82(112)=Qspval3l5*acc82(70)
      acc82(113)=Qspval4l5*acc82(48)
      acc82(114)=Qspvak2l5*acc82(52)
      acc82(115)=QspQ*acc82(6)
      acc82(116)=Qspk2*acc82(16)
      acc82(92)=acc82(92)+acc82(116)+acc82(115)+acc82(114)+acc82(113)+acc82(112&
      &)+acc82(111)+acc82(110)+acc82(109)+acc82(108)+acc82(107)+acc82(106)+acc8&
      &2(105)+acc82(104)+acc82(103)+acc82(102)+acc82(101)+acc82(100)+acc82(99)+&
      &acc82(98)+acc82(97)+acc82(96)+acc82(44)+acc82(95)+acc82(93)+acc82(94)
      acc82(92)=Qspe2*acc82(92)
      acc82(93)=acc82(74)*Qspval3e2
      acc82(94)=acc82(63)*Qspval4e2
      acc82(95)=acc82(54)*Qspvak1e2
      acc82(96)=acc82(51)*Qspvae2l5
      acc82(97)=acc82(45)*Qspvae2k2
      acc82(98)=acc82(29)*Qspvak2e2
      acc82(99)=acc82(28)*Qspvae2k1
      acc82(100)=acc82(14)*Qspvae2l3
      acc82(101)=Qspvak1k2*acc82(10)
      acc82(102)=Qspvak1l3*acc82(34)
      acc82(103)=Qspval3k1*acc82(82)
      acc82(104)=Qspval3k2*acc82(86)
      acc82(105)=Qspval4k1*acc82(77)
      acc82(106)=Qspval4k2*acc82(87)
      acc82(107)=Qspval5k2*acc82(76)
      acc82(108)=Qspval5l3*acc82(71)
      acc82(109)=Qspl5*acc82(69)
      acc82(110)=Qspvak1l5*acc82(32)
      acc82(111)=Qspvak2k1*acc82(3)
      acc82(112)=Qspval3l5*acc82(81)
      acc82(113)=Qspval4l5*acc82(79)
      acc82(114)=Qspvak2l5*acc82(26)
      acc82(115)=QspQ*acc82(47)
      acc82(116)=Qspk2*acc82(17)
      acc82(93)=acc82(116)+acc82(115)+acc82(114)+acc82(113)+acc82(112)+acc82(11&
      &1)+acc82(110)+acc82(109)+acc82(108)+acc82(107)+acc82(106)+acc82(105)+acc&
      &82(104)+acc82(103)+acc82(102)+acc82(101)+acc82(100)+acc82(99)+acc82(98)+&
      &acc82(39)+acc82(97)+acc82(96)+acc82(95)+acc82(93)+acc82(94)
      acc82(93)=Qspe1*acc82(93)
      acc82(94)=Qspvak2l3*acc82(56)
      acc82(95)=Qspvae1e2*acc82(61)
      acc82(96)=Qspvae2e1*acc82(60)
      acc82(97)=-Qspval3l5*acc82(90)
      acc82(98)=-Qspval4l5*acc82(84)
      acc82(94)=acc82(98)+acc82(97)+acc82(96)+acc82(95)+acc82(27)+acc82(94)
      acc82(94)=QspQ*acc82(94)
      acc82(95)=Qspvak2l3*acc82(64)
      acc82(96)=-Qspvae1e2*acc82(31)
      acc82(97)=Qspvae2e1*acc82(13)
      acc82(98)=QspQ*acc82(43)
      acc82(99)=Qspk2*acc82(11)
      acc82(95)=acc82(99)+acc82(98)+acc82(97)+acc82(96)+acc82(15)+acc82(95)
      acc82(95)=Qspk2*acc82(95)
      acc82(96)=Qspval3k2*acc82(22)
      acc82(97)=Qspval4k2*acc82(9)
      acc82(98)=Qspval5k2*acc82(35)
      acc82(99)=Qspval5l3*acc82(56)
      acc82(96)=acc82(99)+acc82(98)+acc82(97)+acc82(38)+acc82(96)
      acc82(96)=Qspvak2l5*acc82(96)
      acc82(97)=Qspval3k1*acc82(90)
      acc82(98)=Qspval4k1*acc82(84)
      acc82(97)=acc82(98)+acc82(53)+acc82(97)
      acc82(97)=Qspvak1l5*acc82(97)
      acc82(98)=-Qspvak1k2*acc82(35)
      acc82(99)=-Qspvak1l3*acc82(56)
      acc82(98)=acc82(99)+acc82(12)+acc82(98)
      acc82(98)=Qspvak2k1*acc82(98)
      acc82(99)=-Qspk1*acc82(8)
      acc82(100)=Qspvak1k2*acc82(2)
      acc82(101)=Qspvak1l3*acc82(36)
      acc82(102)=Qspvak2l3*acc82(65)
      acc82(103)=Qspval3k1*acc82(67)
      acc82(104)=Qspval3k2*acc82(40)
      acc82(105)=Qspval4k1*acc82(55)
      acc82(106)=Qspval4k2*acc82(85)
      acc82(107)=Qspval5k2*acc82(72)
      acc82(108)=Qspval5l3*acc82(59)
      acc82(109)=Qspk1*acc82(31)
      acc82(109)=acc82(42)+acc82(109)
      acc82(109)=Qspvae1e2*acc82(109)
      acc82(110)=-Qspk1*acc82(13)
      acc82(110)=acc82(57)+acc82(110)
      acc82(110)=Qspvae2e1*acc82(110)
      acc82(111)=Qspl5*acc82(4)
      acc82(112)=Qspl5*acc82(89)
      acc82(112)=acc82(78)+acc82(112)
      acc82(112)=Qspval3l5*acc82(112)
      acc82(113)=Qspl5*acc82(83)
      acc82(113)=acc82(50)+acc82(113)
      acc82(113)=Qspval4l5*acc82(113)
      brack=acc82(37)+acc82(92)+acc82(93)+acc82(94)+acc82(95)+acc82(96)+acc82(9&
      &7)+acc82(98)+acc82(99)+acc82(100)+acc82(101)+acc82(102)+acc82(103)+acc82&
      &(104)+acc82(105)+acc82(106)+acc82(107)+acc82(108)+acc82(109)+acc82(110)+&
      &acc82(111)+acc82(112)+acc82(113)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d82h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd82h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d82
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d82 = 0.0_ki
      d82 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d82, ki), aimag(d82), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d82h8l1_qp
