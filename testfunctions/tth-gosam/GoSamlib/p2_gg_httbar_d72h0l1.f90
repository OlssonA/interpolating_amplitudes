module     p2_gg_httbar_d72h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d72h0l1.f90
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
      use p2_gg_httbar_abbrevd72h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc72(51)
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval5e2
      complex(ki) :: QspQ
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2l3
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval5e2 = dotproduct(Q,spval5e2)
      QspQ = dotproduct(Q,Q)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      acc72(1)=abb72(9)
      acc72(2)=abb72(10)
      acc72(3)=abb72(11)
      acc72(4)=abb72(12)
      acc72(5)=abb72(13)
      acc72(6)=abb72(14)
      acc72(7)=abb72(15)
      acc72(8)=abb72(16)
      acc72(9)=abb72(17)
      acc72(10)=abb72(18)
      acc72(11)=abb72(19)
      acc72(12)=abb72(20)
      acc72(13)=abb72(21)
      acc72(14)=abb72(22)
      acc72(15)=abb72(23)
      acc72(16)=abb72(24)
      acc72(17)=abb72(25)
      acc72(18)=abb72(26)
      acc72(19)=abb72(27)
      acc72(20)=abb72(28)
      acc72(21)=abb72(29)
      acc72(22)=abb72(30)
      acc72(23)=abb72(31)
      acc72(24)=abb72(32)
      acc72(25)=abb72(34)
      acc72(26)=abb72(35)
      acc72(27)=abb72(39)
      acc72(28)=abb72(42)
      acc72(29)=abb72(43)
      acc72(30)=abb72(44)
      acc72(31)=abb72(45)
      acc72(32)=abb72(46)
      acc72(33)=abb72(47)
      acc72(34)=abb72(51)
      acc72(35)=abb72(53)
      acc72(36)=abb72(55)
      acc72(37)=abb72(58)
      acc72(38)=abb72(60)
      acc72(39)=abb72(61)
      acc72(40)=acc72(7)*Qspvae2k1
      acc72(41)=acc72(15)*Qspval4e2
      acc72(42)=acc72(16)*Qspvak2e2
      acc72(43)=acc72(22)*Qspvae2k2
      acc72(44)=acc72(25)*Qspvak1e2
      acc72(45)=-acc72(27)*Qspvae2l5
      acc72(46)=acc72(28)*Qspvae1e2
      acc72(47)=-acc72(31)*Qspvae2e1
      acc72(48)=acc72(35)*Qspval5e2
      acc72(40)=acc72(48)+acc72(47)+acc72(46)+acc72(45)+acc72(44)+acc72(23)+acc&
      &72(43)+acc72(42)+acc72(41)+acc72(40)
      acc72(40)=QspQ*acc72(40)
      acc72(41)=acc72(1)*Qspvae1e2
      acc72(42)=acc72(2)*Qspvak2e2
      acc72(43)=acc72(14)*Qspval3e2
      acc72(44)=acc72(17)*Qspvak1e2
      acc72(45)=acc72(19)*Qspval4e2
      acc72(46)=acc72(30)*Qspval5e2
      acc72(41)=acc72(46)+acc72(45)+acc72(44)+acc72(43)+acc72(13)+acc72(41)+acc&
      &72(42)
      acc72(41)=Qspvae2k2*acc72(41)
      acc72(42)=acc72(5)*Qspval4e2
      acc72(43)=acc72(12)*Qspvak2e2
      acc72(44)=acc72(26)*Qspvak1e2
      acc72(45)=acc72(38)*Qspvae1e2
      acc72(46)=acc72(39)*Qspval5e2
      acc72(42)=acc72(46)+acc72(45)+acc72(44)+acc72(43)+acc72(10)+acc72(42)
      acc72(42)=Qspvae2l3*acc72(42)
      acc72(43)=-acc72(9)*Qspvae2k1
      acc72(44)=acc72(36)*Qspvae2e1
      acc72(45)=acc72(37)*Qspvae2l5
      acc72(43)=acc72(45)+acc72(44)+acc72(11)+acc72(43)
      acc72(43)=Qspval4e2*acc72(43)
      acc72(44)=acc72(21)*Qspvae2k1
      acc72(45)=-acc72(33)*Qspvae2e1
      acc72(46)=acc72(34)*Qspvae2l5
      acc72(44)=acc72(46)+acc72(45)+acc72(44)+acc72(18)
      acc72(44)=Qspval3e2*acc72(44)
      acc72(45)=acc72(3)*Qspvak2e2
      acc72(46)=acc72(4)*Qspvae2k1
      acc72(47)=acc72(6)*Qspvae2l5
      acc72(48)=acc72(20)*Qspvak1e2
      acc72(49)=acc72(24)*Qspvae1e2
      acc72(50)=acc72(29)*Qspvae2e1
      acc72(51)=acc72(32)*Qspval5e2
      brack=acc72(8)+acc72(40)+acc72(41)+acc72(42)+acc72(43)+acc72(44)+acc72(45&
      &)+acc72(46)+acc72(47)+acc72(48)+acc72(49)+acc72(50)+acc72(51)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d72h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd72h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d72
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d72 = 0.0_ki
      d72 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d72, ki), aimag(d72), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d72h0l1
