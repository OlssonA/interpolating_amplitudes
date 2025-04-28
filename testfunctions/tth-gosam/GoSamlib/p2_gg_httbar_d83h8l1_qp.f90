module     p2_gg_httbar_d83h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d83h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd83h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc83(53)
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: QspQ
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      QspQ = dotproduct(Q,Q)
      acc83(1)=abb83(8)
      acc83(2)=abb83(9)
      acc83(3)=abb83(10)
      acc83(4)=abb83(11)
      acc83(5)=abb83(12)
      acc83(6)=abb83(13)
      acc83(7)=abb83(14)
      acc83(8)=abb83(15)
      acc83(9)=abb83(16)
      acc83(10)=abb83(17)
      acc83(11)=abb83(18)
      acc83(12)=abb83(19)
      acc83(13)=abb83(20)
      acc83(14)=abb83(21)
      acc83(15)=abb83(22)
      acc83(16)=abb83(23)
      acc83(17)=abb83(24)
      acc83(18)=abb83(25)
      acc83(19)=abb83(26)
      acc83(20)=abb83(27)
      acc83(21)=abb83(28)
      acc83(22)=abb83(29)
      acc83(23)=abb83(30)
      acc83(24)=abb83(36)
      acc83(25)=abb83(40)
      acc83(26)=abb83(41)
      acc83(27)=abb83(43)
      acc83(28)=abb83(45)
      acc83(29)=abb83(48)
      acc83(30)=abb83(49)
      acc83(31)=abb83(51)
      acc83(32)=abb83(53)
      acc83(33)=abb83(54)
      acc83(34)=abb83(56)
      acc83(35)=-acc83(9)*Qspval4e2
      acc83(36)=-acc83(13)*Qspval3e2
      acc83(35)=acc83(30)+acc83(36)+acc83(35)
      acc83(35)=acc83(35)*Qspvae1l5
      acc83(36)=acc83(5)*Qspval3e2
      acc83(37)=acc83(20)*Qspvae1k2
      acc83(38)=acc83(24)*Qspval4e2
      acc83(39)=acc83(25)*Qspvae1l3
      acc83(35)=acc83(38)+acc83(36)+acc83(35)+acc83(39)+acc83(37)+acc83(14)
      acc83(35)=Qspvae2e1*acc83(35)
      acc83(36)=acc83(4)*Qspvae2k2
      acc83(37)=acc83(12)*Qspvae2l3
      acc83(36)=acc83(37)+acc83(36)+acc83(3)
      acc83(36)=acc83(36)*Qspvak2e1
      acc83(37)=acc83(11)*Qspvae2k2
      acc83(38)=acc83(28)*Qspvae2l3
      acc83(39)=acc83(32)*Qspval4e1
      acc83(40)=acc83(34)*Qspval3e1
      acc83(36)=acc83(38)+acc83(37)+acc83(36)+acc83(40)+acc83(39)+acc83(23)
      acc83(36)=Qspvae1e2*acc83(36)
      acc83(37)=acc83(1)*Qspval4e2
      acc83(38)=acc83(6)*Qspval3e1
      acc83(39)=acc83(10)*Qspvak2e1
      acc83(40)=acc83(17)*Qspvae2k2
      acc83(41)=acc83(19)*Qspvae1k2
      acc83(42)=acc83(22)*Qspvae2l3
      acc83(43)=acc83(26)*Qspvae1l3
      acc83(44)=acc83(29)*Qspvae1l5
      acc83(45)=acc83(31)*Qspval4e1
      acc83(46)=acc83(33)*Qspval3e2
      acc83(47)=Qspval4k1*acc83(7)
      acc83(48)=Qspval3k1*acc83(15)
      acc83(49)=Qspvak2k1*acc83(8)
      acc83(50)=Qspvak1l5*acc83(16)
      acc83(51)=Qspvak1l3*acc83(27)
      acc83(52)=Qspvak1k2*acc83(21)
      acc83(53)=QspQ*acc83(18)
      brack=acc83(2)+acc83(35)+acc83(36)+acc83(37)+acc83(38)+acc83(39)+acc83(40&
      &)+acc83(41)+acc83(42)+acc83(43)+acc83(44)+acc83(45)+acc83(46)+acc83(47)+&
      &acc83(48)+acc83(49)+acc83(50)+acc83(51)+acc83(52)+acc83(53)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d83h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd83h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d83
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d83 = 0.0_ki
      d83 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d83, ki), aimag(d83), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d83h8l1_qp
