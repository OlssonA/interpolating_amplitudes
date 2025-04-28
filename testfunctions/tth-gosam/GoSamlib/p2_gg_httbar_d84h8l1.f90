module     p2_gg_httbar_d84h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d84h8l1.f90
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
      use p2_gg_httbar_abbrevd84h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc84(90)
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspe2
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspk1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspe2 = dotproduct(Q,e2)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspk1 = dotproduct(Q,k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      acc84(1)=abb84(8)
      acc84(2)=abb84(9)
      acc84(3)=abb84(10)
      acc84(4)=abb84(11)
      acc84(5)=abb84(12)
      acc84(6)=abb84(13)
      acc84(7)=abb84(14)
      acc84(8)=abb84(15)
      acc84(9)=abb84(16)
      acc84(10)=abb84(17)
      acc84(11)=abb84(18)
      acc84(12)=abb84(19)
      acc84(13)=abb84(20)
      acc84(14)=abb84(21)
      acc84(15)=abb84(22)
      acc84(16)=abb84(23)
      acc84(17)=abb84(24)
      acc84(18)=abb84(25)
      acc84(19)=abb84(26)
      acc84(20)=abb84(27)
      acc84(21)=abb84(28)
      acc84(22)=abb84(29)
      acc84(23)=abb84(30)
      acc84(24)=abb84(31)
      acc84(25)=abb84(32)
      acc84(26)=abb84(33)
      acc84(27)=abb84(34)
      acc84(28)=abb84(35)
      acc84(29)=abb84(36)
      acc84(30)=abb84(37)
      acc84(31)=abb84(38)
      acc84(32)=abb84(39)
      acc84(33)=abb84(40)
      acc84(34)=abb84(41)
      acc84(35)=abb84(42)
      acc84(36)=abb84(43)
      acc84(37)=abb84(44)
      acc84(38)=abb84(46)
      acc84(39)=abb84(48)
      acc84(40)=abb84(50)
      acc84(41)=abb84(51)
      acc84(42)=abb84(53)
      acc84(43)=abb84(55)
      acc84(44)=abb84(56)
      acc84(45)=abb84(57)
      acc84(46)=abb84(58)
      acc84(47)=abb84(60)
      acc84(48)=abb84(61)
      acc84(49)=abb84(62)
      acc84(50)=abb84(63)
      acc84(51)=abb84(64)
      acc84(52)=abb84(65)
      acc84(53)=abb84(66)
      acc84(54)=abb84(67)
      acc84(55)=abb84(68)
      acc84(56)=abb84(69)
      acc84(57)=abb84(70)
      acc84(58)=abb84(71)
      acc84(59)=abb84(73)
      acc84(60)=abb84(74)
      acc84(61)=abb84(75)
      acc84(62)=abb84(76)
      acc84(63)=abb84(77)
      acc84(64)=abb84(78)
      acc84(65)=abb84(79)
      acc84(66)=abb84(80)
      acc84(67)=abb84(81)
      acc84(68)=abb84(82)
      acc84(69)=abb84(86)
      acc84(70)=abb84(87)
      acc84(71)=abb84(89)
      acc84(72)=acc84(15)*Qspval4l5
      acc84(73)=acc84(25)*Qspvak2l3
      acc84(74)=acc84(38)*Qspk2
      acc84(75)=acc84(61)*Qspval3l5
      acc84(72)=acc84(75)+acc84(74)+acc84(73)+acc84(72)+acc84(4)
      acc84(72)=acc84(72)*Qspe2
      acc84(73)=acc84(12)*Qspl5
      acc84(74)=acc84(14)*QspQ
      acc84(75)=acc84(21)*Qspk2
      acc84(76)=acc84(24)*Qspvak2l5
      acc84(77)=-acc84(28)*Qspval5l3
      acc84(78)=acc84(49)*Qspval4l5
      acc84(79)=acc84(52)*Qspval3k2
      acc84(80)=acc84(59)*Qspval3l5
      acc84(81)=acc84(63)*Qspval4k2
      acc84(82)=-acc84(69)*Qspval5k2
      acc84(83)=Qspvae2l5*acc84(50)
      acc84(84)=Qspval4e2*acc84(56)
      acc84(85)=Qspvae2l3*acc84(66)
      acc84(86)=Qspval3e2*acc84(55)
      acc84(87)=Qspvae2k2*acc84(54)
      acc84(88)=Qspvak2e2*acc84(65)
      acc84(89)=Qspvae2k1*acc84(35)
      acc84(90)=Qspvak1e2*acc84(33)
      acc84(72)=acc84(80)+acc84(78)+acc84(75)+acc84(72)+acc84(90)+acc84(89)+acc&
      &84(88)+acc84(87)+acc84(86)+acc84(85)+acc84(84)+acc84(83)+acc84(82)+acc84&
      &(81)+acc84(79)+acc84(77)+acc84(76)+acc84(74)+acc84(73)+acc84(1)
      acc84(72)=Qspe1*acc84(72)
      acc84(73)=acc84(7)*QspQ
      acc84(74)=acc84(16)*Qspvak2l5
      acc84(75)=acc84(17)*Qspk2
      acc84(76)=acc84(32)*Qspl5
      acc84(77)=acc84(43)*Qspval5l3
      acc84(78)=acc84(45)*Qspval5k2
      acc84(79)=acc84(60)*Qspval4k2
      acc84(80)=acc84(62)*Qspval3k2
      acc84(81)=acc84(70)*Qspval3l5
      acc84(82)=acc84(71)*Qspval4l5
      acc84(83)=Qspvae1l5*acc84(53)
      acc84(84)=Qspval4e1*acc84(64)
      acc84(85)=Qspvae1l3*acc84(67)
      acc84(86)=Qspval3e1*acc84(68)
      acc84(87)=-Qspvae1k2*acc84(57)
      acc84(88)=Qspvak2e1*acc84(11)
      acc84(89)=Qspvae1k1*acc84(41)
      acc84(90)=Qspvak1e1*acc84(30)
      acc84(73)=acc84(82)+acc84(81)+acc84(75)+acc84(90)+acc84(89)+acc84(88)+acc&
      &84(87)+acc84(86)+acc84(85)+acc84(84)+acc84(83)+acc84(80)+acc84(79)+acc84&
      &(78)+acc84(44)+acc84(77)+acc84(76)+acc84(74)+acc84(73)
      acc84(73)=Qspe2*acc84(73)
      acc84(74)=acc84(20)*Qspk2
      acc84(75)=acc84(36)*Qspvae2e1
      acc84(76)=acc84(42)*Qspval3l5
      acc84(77)=acc84(47)*Qspvae1e2
      acc84(78)=acc84(58)*Qspval4l5
      acc84(74)=acc84(78)+acc84(77)+acc84(76)+acc84(75)+acc84(74)+acc84(2)
      acc84(74)=QspQ*acc84(74)
      acc84(75)=acc84(6)*Qspval4k2
      acc84(76)=acc84(8)*Qspval3k2
      acc84(77)=acc84(19)*Qspval5k2
      acc84(75)=acc84(77)+acc84(13)+acc84(76)+acc84(75)
      acc84(75)=Qspvak2l5*acc84(75)
      acc84(76)=-acc84(23)*Qspvae2e1
      acc84(77)=acc84(37)*Qspvae1e2
      acc84(76)=acc84(77)+acc84(76)
      acc84(77)=Qspk1-Qspk2
      acc84(76)=acc84(77)*acc84(76)
      acc84(77)=acc84(34)*Qspval3l5
      acc84(78)=acc84(51)*Qspval4l5
      acc84(77)=acc84(78)+acc84(77)+acc84(10)
      acc84(77)=Qspl5*acc84(77)
      acc84(78)=Qspvak2l3*QspQ
      acc84(79)=Qspval5l3*Qspvak2l5
      acc84(78)=acc84(78)+acc84(79)
      acc84(78)=acc84(26)*acc84(78)
      acc84(79)=acc84(22)*Qspk2
      acc84(79)=acc84(79)+acc84(5)
      acc84(79)=Qspvak2l3*acc84(79)
      acc84(80)=acc84(3)*Qspk2**2
      acc84(81)=acc84(9)*Qspk2
      acc84(82)=acc84(27)*Qspvae2e1
      acc84(83)=acc84(29)*Qspk1
      acc84(84)=acc84(40)*Qspvae1e2
      acc84(85)=acc84(46)*Qspval3l5
      acc84(86)=acc84(48)*Qspval4l5
      acc84(87)=Qspvak2k1*acc84(18)
      acc84(88)=Qspvak1l5*acc84(31)
      brack=acc84(39)+acc84(72)+acc84(73)+acc84(74)+acc84(75)+acc84(76)+acc84(7&
      &7)+acc84(78)+acc84(79)+acc84(80)+acc84(81)+acc84(82)+acc84(83)+acc84(84)&
      &+acc84(85)+acc84(86)+acc84(87)+acc84(88)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d84h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd84h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d84
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d84 = 0.0_ki
      d84 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d84, ki), aimag(d84), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d84h8l1
