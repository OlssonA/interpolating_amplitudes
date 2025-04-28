module     p2_gg_httbar_d91h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d91h12l1.f90
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
      use p2_gg_httbar_abbrevd91h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc91(57)
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvae1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspe2
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      QspQ = dotproduct(Q,Q)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspe2 = dotproduct(Q,e2)
      acc91(1)=abb91(8)
      acc91(2)=abb91(9)
      acc91(3)=abb91(10)
      acc91(4)=abb91(11)
      acc91(5)=abb91(12)
      acc91(6)=abb91(13)
      acc91(7)=abb91(14)
      acc91(8)=abb91(15)
      acc91(9)=abb91(16)
      acc91(10)=abb91(17)
      acc91(11)=abb91(18)
      acc91(12)=abb91(19)
      acc91(13)=abb91(20)
      acc91(14)=abb91(21)
      acc91(15)=abb91(22)
      acc91(16)=abb91(23)
      acc91(17)=abb91(24)
      acc91(18)=abb91(25)
      acc91(19)=abb91(26)
      acc91(20)=abb91(27)
      acc91(21)=abb91(28)
      acc91(22)=abb91(29)
      acc91(23)=abb91(30)
      acc91(24)=abb91(31)
      acc91(25)=abb91(32)
      acc91(26)=abb91(33)
      acc91(27)=abb91(34)
      acc91(28)=abb91(35)
      acc91(29)=abb91(37)
      acc91(30)=abb91(42)
      acc91(31)=abb91(44)
      acc91(32)=abb91(47)
      acc91(33)=abb91(49)
      acc91(34)=abb91(50)
      acc91(35)=abb91(55)
      acc91(36)=abb91(57)
      acc91(37)=abb91(61)
      acc91(38)=abb91(65)
      acc91(39)=abb91(67)
      acc91(40)=abb91(69)
      acc91(41)=abb91(74)
      acc91(42)=abb91(76)
      acc91(43)=acc91(2)*Qspvae1l4
      acc91(44)=acc91(16)*Qspvae1l5
      acc91(45)=acc91(22)*Qspval3k1
      acc91(46)=acc91(24)*Qspvak2k1
      acc91(47)=acc91(31)*Qspvae1k2
      acc91(48)=acc91(33)*QspQ
      acc91(49)=acc91(38)*Qspvae1l3
      acc91(43)=acc91(49)+acc91(48)+acc91(47)+acc91(26)+acc91(46)+acc91(45)+acc&
      &91(44)+acc91(43)
      acc91(43)=Qspvae2e1*acc91(43)
      acc91(44)=acc91(4)*Qspvak2e1
      acc91(45)=acc91(25)*Qspvak1l5
      acc91(46)=acc91(28)*Qspvak1l3
      acc91(47)=acc91(29)*Qspval3e1
      acc91(48)=acc91(36)*QspQ
      acc91(49)=acc91(37)*Qspval4e1
      acc91(44)=acc91(49)+acc91(48)+acc91(47)+acc91(46)+acc91(45)+acc91(44)+acc&
      &91(1)
      acc91(44)=Qspvae1e2*acc91(44)
      acc91(45)=acc91(7)*Qspvae1l5
      acc91(46)=acc91(15)*Qspvae1l4
      acc91(47)=acc91(19)*Qspvae1l3
      acc91(45)=acc91(47)+acc91(46)+acc91(11)+acc91(45)
      acc91(45)=acc91(45)*Qspvak2e1
      acc91(46)=acc91(42)*Qspval3e1
      acc91(46)=acc91(46)+acc91(9)
      acc91(46)=acc91(46)*Qspvae1l4
      acc91(47)=acc91(30)*Qspvae1l5
      acc91(48)=acc91(32)*Qspvae1l3
      acc91(49)=acc91(34)*Qspval3e1
      acc91(45)=acc91(49)+acc91(48)+acc91(47)+acc91(13)+acc91(45)+acc91(46)
      acc91(45)=Qspe2*acc91(45)
      acc91(46)=acc91(5)*Qspvae1l5
      acc91(47)=acc91(8)*Qspvae1l4
      acc91(48)=acc91(18)*Qspvae1l3
      acc91(46)=acc91(48)+acc91(47)+acc91(6)+acc91(46)
      acc91(46)=Qspvak2e1*acc91(46)
      acc91(47)=acc91(41)*Qspvae1l4
      acc91(47)=acc91(47)+acc91(40)
      acc91(47)=Qspval3e1*acc91(47)
      acc91(48)=acc91(3)*Qspvae1l4
      acc91(49)=acc91(10)*Qspvae1k2
      acc91(50)=acc91(14)*QspQ
      acc91(51)=acc91(17)*Qspval4e1
      acc91(52)=acc91(20)*Qspvak1l5
      acc91(53)=acc91(21)*Qspval3k1
      acc91(54)=acc91(23)*Qspvak2k1
      acc91(55)=acc91(27)*Qspvak1l3
      acc91(56)=acc91(35)*Qspvae1l3
      acc91(57)=acc91(39)*Qspvae1l5
      brack=acc91(12)+acc91(43)+acc91(44)+acc91(45)+acc91(46)+acc91(47)+acc91(4&
      &8)+acc91(49)+acc91(50)+acc91(51)+acc91(52)+acc91(53)+acc91(54)+acc91(55)&
      &+acc91(56)+acc91(57)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d91h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd91h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d91
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d91 = 0.0_ki
      d91 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d91, ki), aimag(d91), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d91h12l1
