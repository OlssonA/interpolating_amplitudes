module     p2_gg_httbar_d36h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d36h0l1.f90
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
      use p2_gg_httbar_abbrevd36h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc36(41)
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspl4
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspl4 = dotproduct(Q,l4)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      acc36(1)=abb36(15)
      acc36(2)=abb36(16)
      acc36(3)=abb36(17)
      acc36(4)=abb36(19)
      acc36(5)=abb36(20)
      acc36(6)=abb36(21)
      acc36(7)=abb36(22)
      acc36(8)=abb36(23)
      acc36(9)=abb36(24)
      acc36(10)=abb36(26)
      acc36(11)=abb36(27)
      acc36(12)=abb36(29)
      acc36(13)=abb36(30)
      acc36(14)=abb36(31)
      acc36(15)=abb36(34)
      acc36(16)=abb36(35)
      acc36(17)=abb36(38)
      acc36(18)=abb36(40)
      acc36(19)=abb36(42)
      acc36(20)=abb36(43)
      acc36(21)=abb36(70)
      acc36(22)=Qspval4e2*acc36(8)
      acc36(23)=Qspval4e1*acc36(13)
      acc36(24)=Qspvae2l3*acc36(17)
      acc36(25)=Qspval3e2*acc36(18)
      acc36(26)=Qspvae1l3*acc36(19)
      acc36(27)=Qspval3e1*acc36(20)
      acc36(28)=Qspvae2k2*acc36(12)
      acc36(29)=Qspvae1k2*acc36(16)
      acc36(30)=Qspval4l3*acc36(15)
      acc36(31)=Qspval4k2*acc36(2)
      acc36(32)=Qspval4k1*acc36(6)
      acc36(33)=Qspval3l4*acc36(10)
      acc36(34)=Qspval3k2*acc36(5)
      acc36(35)=Qspval3k1*acc36(7)
      acc36(36)=Qspvak2l3*acc36(11)
      acc36(37)=Qspvak1l3*acc36(14)
      acc36(38)=Qspvak1k2*acc36(9)
      acc36(39)=Qspl4*acc36(21)
      acc36(40)=Qspl3*acc36(1)
      acc36(41)=Qspk2*acc36(3)
      brack=acc36(4)+acc36(22)+acc36(23)+acc36(24)+acc36(25)+acc36(26)+acc36(27&
      &)+acc36(28)+acc36(29)+acc36(30)+acc36(31)+acc36(32)+acc36(33)+acc36(34)+&
      &acc36(35)+acc36(36)+acc36(37)+acc36(38)+acc36(39)+acc36(40)+acc36(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d36h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd36h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d36
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d36 = 0.0_ki
      d36 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d36, ki), aimag(d36), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d36h0l1
