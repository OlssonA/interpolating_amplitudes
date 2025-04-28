module     p2_gg_httbar_d70h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d70h0l1.f90
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
      use p2_gg_httbar_abbrevd70h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc70(51)
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: QspQ
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      QspQ = dotproduct(Q,Q)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      acc70(1)=abb70(9)
      acc70(2)=abb70(10)
      acc70(3)=abb70(11)
      acc70(4)=abb70(12)
      acc70(5)=abb70(13)
      acc70(6)=abb70(14)
      acc70(7)=abb70(15)
      acc70(8)=abb70(16)
      acc70(9)=abb70(17)
      acc70(10)=abb70(18)
      acc70(11)=abb70(19)
      acc70(12)=abb70(20)
      acc70(13)=abb70(21)
      acc70(14)=abb70(22)
      acc70(15)=abb70(23)
      acc70(16)=abb70(24)
      acc70(17)=abb70(25)
      acc70(18)=abb70(26)
      acc70(19)=abb70(27)
      acc70(20)=abb70(28)
      acc70(21)=abb70(29)
      acc70(22)=abb70(30)
      acc70(23)=abb70(31)
      acc70(24)=abb70(32)
      acc70(25)=abb70(33)
      acc70(26)=abb70(34)
      acc70(27)=abb70(36)
      acc70(28)=abb70(39)
      acc70(29)=abb70(40)
      acc70(30)=abb70(41)
      acc70(31)=abb70(43)
      acc70(32)=abb70(45)
      acc70(33)=abb70(47)
      acc70(34)=abb70(49)
      acc70(35)=abb70(53)
      acc70(36)=abb70(54)
      acc70(37)=abb70(56)
      acc70(38)=abb70(58)
      acc70(39)=abb70(59)
      acc70(40)=acc70(1)*Qspval5e2
      acc70(41)=acc70(12)*Qspvae2k1
      acc70(42)=acc70(18)*Qspvae2k2
      acc70(43)=acc70(23)*Qspvak1e2
      acc70(44)=acc70(27)*Qspvak2e2
      acc70(45)=acc70(30)*Qspvae2e1
      acc70(46)=acc70(34)*Qspvae1e2
      acc70(47)=acc70(35)*Qspvae2l4
      acc70(48)=acc70(37)*Qspval4e2
      acc70(40)=acc70(48)+acc70(47)+acc70(46)+acc70(45)+acc70(44)+acc70(43)+acc&
      &70(21)+acc70(42)+acc70(41)+acc70(40)
      acc70(40)=QspQ*acc70(40)
      acc70(41)=acc70(2)*Qspvak2e2
      acc70(42)=acc70(6)*Qspval5e2
      acc70(43)=acc70(11)*Qspvae1e2
      acc70(44)=acc70(16)*Qspvak1e2
      acc70(45)=acc70(20)*Qspval3e2
      acc70(46)=acc70(25)*Qspval4e2
      acc70(41)=acc70(46)+acc70(45)+acc70(44)+acc70(43)+acc70(7)+acc70(42)+acc7&
      &0(41)
      acc70(41)=Qspvae2k2*acc70(41)
      acc70(42)=acc70(15)*Qspvak2e2
      acc70(43)=acc70(24)*Qspvak1e2
      acc70(44)=acc70(28)*Qspvae1e2
      acc70(45)=acc70(31)*Qspval5e2
      acc70(46)=acc70(38)*Qspval4e2
      acc70(42)=acc70(46)+acc70(45)+acc70(44)+acc70(43)+acc70(17)+acc70(42)
      acc70(42)=Qspvae2l3*acc70(42)
      acc70(43)=acc70(13)*Qspvae2l4
      acc70(44)=acc70(19)*Qspvae2k1
      acc70(45)=acc70(26)*Qspvae2e1
      acc70(43)=acc70(45)+acc70(44)+acc70(43)+acc70(8)
      acc70(43)=Qspval5e2*acc70(43)
      acc70(44)=acc70(5)*Qspvae2k1
      acc70(45)=acc70(29)*Qspvae2e1
      acc70(46)=acc70(39)*Qspvae2l4
      acc70(44)=acc70(46)+acc70(45)+acc70(22)+acc70(44)
      acc70(44)=Qspval3e2*acc70(44)
      acc70(45)=acc70(3)*Qspvae2k1
      acc70(46)=acc70(9)*Qspvae2e1
      acc70(47)=acc70(10)*Qspvak1e2
      acc70(48)=acc70(14)*Qspvak2e2
      acc70(49)=acc70(32)*Qspvae2l4
      acc70(50)=acc70(33)*Qspvae1e2
      acc70(51)=acc70(36)*Qspval4e2
      brack=acc70(4)+acc70(40)+acc70(41)+acc70(42)+acc70(43)+acc70(44)+acc70(45&
      &)+acc70(46)+acc70(47)+acc70(48)+acc70(49)+acc70(50)+acc70(51)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d70h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd70h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d70
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d70 = 0.0_ki
      d70 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d70, ki), aimag(d70), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d70h0l1
