module     p2_gg_httbar_d81h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d81h8l1.f90
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
      use p2_gg_httbar_abbrevd81h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc81(49)
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc81(1)=abb81(8)
      acc81(2)=abb81(9)
      acc81(3)=abb81(10)
      acc81(4)=abb81(11)
      acc81(5)=abb81(12)
      acc81(6)=abb81(13)
      acc81(7)=abb81(14)
      acc81(8)=abb81(15)
      acc81(9)=abb81(16)
      acc81(10)=abb81(17)
      acc81(11)=abb81(18)
      acc81(12)=abb81(19)
      acc81(13)=abb81(20)
      acc81(14)=abb81(21)
      acc81(15)=abb81(22)
      acc81(16)=abb81(23)
      acc81(17)=abb81(24)
      acc81(18)=abb81(30)
      acc81(19)=abb81(31)
      acc81(20)=abb81(38)
      acc81(21)=abb81(39)
      acc81(22)=abb81(41)
      acc81(23)=abb81(42)
      acc81(24)=abb81(43)
      acc81(25)=abb81(44)
      acc81(26)=abb81(45)
      acc81(27)=abb81(46)
      acc81(28)=abb81(47)
      acc81(29)=abb81(48)
      acc81(30)=abb81(51)
      acc81(31)=abb81(56)
      acc81(32)=-acc81(3)*Qspval4e1
      acc81(33)=-acc81(7)*Qspval3e1
      acc81(32)=acc81(19)+acc81(33)+acc81(32)
      acc81(32)=acc81(32)*Qspvae2l5
      acc81(33)=acc81(4)*Qspvae2k2
      acc81(34)=acc81(20)*Qspval3e1
      acc81(35)=acc81(24)*Qspvae2l3
      acc81(36)=acc81(28)*Qspval4e1
      acc81(32)=acc81(36)+acc81(34)+acc81(32)+acc81(35)+acc81(17)+acc81(33)
      acc81(32)=Qspvae1e2*acc81(32)
      acc81(33)=acc81(1)*Qspvae1l3
      acc81(34)=acc81(5)*Qspvae1k2
      acc81(33)=acc81(18)+acc81(34)+acc81(33)
      acc81(33)=acc81(33)*Qspvae2e1
      acc81(33)=acc81(10)+acc81(33)
      acc81(33)=Qspvak2e2*acc81(33)
      acc81(34)=Qspval4e2*acc81(27)
      acc81(35)=Qspval3e2*acc81(6)
      acc81(34)=acc81(35)+acc81(34)+acc81(8)
      acc81(34)=Qspvae2e1*acc81(34)
      acc81(35)=acc81(21)*Qspvae2k2
      acc81(36)=acc81(22)*Qspval3e1
      acc81(37)=acc81(23)*Qspvae1l3
      acc81(38)=acc81(25)*Qspvae2l5
      acc81(39)=acc81(26)*Qspval4e1
      acc81(40)=acc81(29)*Qspvae2l3
      acc81(41)=-acc81(30)*Qspvae1k2
      acc81(42)=Qspvae2k1*acc81(9)
      acc81(43)=Qspvak1e2*acc81(13)
      acc81(44)=Qspval4k2*acc81(11)
      acc81(45)=Qspval3k2*acc81(31)
      acc81(46)=Qspvak2l5*acc81(14)
      acc81(47)=Qspvak2l3*acc81(16)
      acc81(48)=Qspk2*acc81(12)
      acc81(49)=QspQ*acc81(15)
      brack=acc81(2)+acc81(32)+acc81(33)+acc81(34)+acc81(35)+acc81(36)+acc81(37&
      &)+acc81(38)+acc81(39)+acc81(40)+acc81(41)+acc81(42)+acc81(43)+acc81(44)+&
      &acc81(45)+acc81(46)+acc81(47)+acc81(48)+acc81(49)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d81h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd81h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d81
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d81 = 0.0_ki
      d81 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d81, ki), aimag(d81), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d81h8l1
